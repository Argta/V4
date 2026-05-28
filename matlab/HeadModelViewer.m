function HeadModelViewer()
    % Simple GUI to display MaleHead.obj 3D model

    objPath = fullfile(fileparts(mfilename('fullpath')), 'MaleHead.obj');

    [V, F] = readOBJ(objPath);

    % Create figure
    fig = figure('Name', 'Male Head Model Viewer', ...
                 'Position', [200, 100, 900, 700], ...
                 'NumberTitle', 'off', ...
                 'ToolBar', 'figure');

    % 3D axes
    ax = axes('Parent', fig, 'Units', 'pixels', ...
              'Position', [50, 80, 800, 600]);
    title(ax, 'Male Head Model', 'FontSize', 14);
    xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
    axis(ax, 'equal');
    grid(ax, 'on');
    hold(ax, 'on');
    view(ax, 3);
    cameratoolbar('Show');

    % Render mesh
    patch('Parent', ax, 'Faces', F, 'Vertices', V, ...
          'FaceColor', [0.8 0.7 0.6], ...
          'EdgeColor', 'none', ...
          'FaceLighting', 'gouraud', ...
          'AmbientStrength', 0.5);

    % Lighting
    light('Position', [1, 0, 0], 'Style', 'infinite');
    light('Position', [-1, 0, 0], 'Style', 'infinite');
    light('Position', [0, 1, 1], 'Style', 'infinite');

    % Buttons
    uicontrol('Style', 'pushbutton', 'String', 'Front View', ...
              'Position', [60, 30, 90, 30], ...
              'Callback', @(src,evt) view(ax, [0, 0, 1]));

    uicontrol('Style', 'pushbutton', 'String', 'Side View', ...
              'Position', [165, 30, 90, 30], ...
              'Callback', @(src,evt) view(ax, [1, 0, 0]));

    uicontrol('Style', 'pushbutton', 'String', 'Top View', ...
              'Position', [270, 30, 90, 30], ...
              'Callback', @(src,evt) view(ax, [0, 1, 0]));

    uicontrol('Style', 'pushbutton', 'String', 'Reset View', ...
              'Position', [375, 30, 90, 30], ...
              'Callback', @(src,evt) view(ax, 3));

    uicontrol('Style', 'text', ...
              'String', sprintf('%d vertices, %d faces', size(V,1), size(F,1)), ...
              'Position', [520, 35, 200, 22], ...
              'BackgroundColor', get(fig, 'Color'));
end

function [V, F] = readOBJ(filepath)
    fid = fopen(filepath, 'r');
    if fid == -1
        error('Cannot open file: %s', filepath);
    end

    V = [];
    F = [];

    while ~feof(fid)
        line = fgetl(fid);
        if isempty(line) || line(1) == '#' || line(1) == 'm' || line(1) == 'u'
            continue;
        end

        tokens = strsplit(strtrim(line));
        if isempty(tokens{1})
            continue;
        end

        if tokens{1} == 'v'
            V(end+1, :) = [str2double(tokens{2}), ...
                           str2double(tokens{3}), ...
                           str2double(tokens{4})];
        elseif tokens{1} == 'f'
            n = numel(tokens) - 1;
            faceVerts = zeros(1, n);
            for i = 2:numel(tokens)
                parts = strsplit(tokens{i}, '/');
                faceVerts(i-1) = str2double(parts{1});
            end
            if n == 3
                F(end+1, :) = faceVerts;
            elseif n == 4
                F(end+1, :) = faceVerts([1, 2, 3]);
                F(end+1, :) = faceVerts([1, 3, 4]);
            elseif n > 4
                for k = 3:n
                    F(end+1, :) = faceVerts([1, k-1, k]);
                end
            end
        end
    end

    fclose(fid);
end
